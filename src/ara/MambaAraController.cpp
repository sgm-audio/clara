#include "MambaAraController.hpp"
#include <cstring>
#include <dlfcn.h>

// -----------------------------------------------------------------------------
// Model Path Resolution (same strategy as CLAP plugin — via dladdr)
// -----------------------------------------------------------------------------

static std::string resolveModelDirOnce() {
    Dl_info info;
    if (dladdr((void*)resolveModelDirOnce, &info) && info.dli_fname) {
        std::string path(info.dli_fname);
        auto sep = path.rfind('/');
        if (sep != std::string::npos)
            path.resize(sep + 1);
        else
            return "./";
        return path + "models/";
    }
    return "./models/";
}

const std::string& getMambaModelDir() {
    static const std::string dir = resolveModelDirOnce();
    return dir;
}

// -----------------------------------------------------------------------------
// PlaybackRenderer
// -----------------------------------------------------------------------------

MambaPlaybackRenderer::MambaPlaybackRenderer(ARA::PlugIn::DocumentController* documentController) noexcept
: ARA::PlugIn::PlaybackRenderer(documentController)
{
    MambaInferenceEngine::Config config;
    config.modelPath = getMambaModelDir() + "mamba_rt.onnx";
    config.maxBlockSize = 1024;
    rtEngine_ = std::make_unique<MambaInferenceEngine>(config);
    try {
        rtEngine_->init();
        engineValid_ = true;
    } catch (const std::exception&) {
        engineValid_ = false;
    }
}

MambaPlaybackRenderer::~MambaPlaybackRenderer() noexcept = default;

void MambaPlaybackRenderer::willAddPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    ARA::PlugIn::PlaybackRenderer::willAddPlaybackRegion(playbackRegion);
}

void MambaPlaybackRenderer::didAddPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    ARA::PlugIn::PlaybackRenderer::didAddPlaybackRegion(playbackRegion);
}

void MambaPlaybackRenderer::willRemovePlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    ARA::PlugIn::PlaybackRenderer::willRemovePlaybackRegion(playbackRegion);
}

void MambaPlaybackRenderer::didRemovePlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    ARA::PlugIn::PlaybackRenderer::didRemovePlaybackRegion(playbackRegion);
}

// -----------------------------------------------------------------------------
// DocumentController
// -----------------------------------------------------------------------------

ARA::PlugIn::Document* MambaDocumentController::doCreateDocument() noexcept
{
    return new ARA::PlugIn::Document(this);
}

void MambaDocumentController::doDestroyDocument(ARA::PlugIn::Document* document) noexcept
{
    delete document;
}

ARA::PlugIn::MusicalContext* MambaDocumentController::doCreateMusicalContext(ARA::PlugIn::Document* document, ARA::ARAMusicalContextHostRef hostRef) noexcept
{
    return new ARA::PlugIn::MusicalContext(document, hostRef);
}

void MambaDocumentController::doDestroyMusicalContext(ARA::PlugIn::MusicalContext* musicalContext) noexcept
{
    delete musicalContext;
}

void MambaDocumentController::doUpdateMusicalContextContent(ARA::PlugIn::MusicalContext*, const ARA::ARAContentTimeRange*, ARA::ContentUpdateScopes) noexcept
{
}

ARA::PlugIn::RegionSequence* MambaDocumentController::doCreateRegionSequence(ARA::PlugIn::Document* document, ARA::ARARegionSequenceHostRef hostRef) noexcept
{
    return new ARA::PlugIn::RegionSequence(document, hostRef);
}

void MambaDocumentController::doDestroyRegionSequence(ARA::PlugIn::RegionSequence* regionSequence) noexcept
{
    delete regionSequence;
}

ARA::PlugIn::AudioSource* MambaDocumentController::doCreateAudioSource(ARA::PlugIn::Document* document, ARA::ARAAudioSourceHostRef hostRef) noexcept
{
    return new ARA::PlugIn::AudioSource(document, hostRef);
}

void MambaDocumentController::doDestroyAudioSource(ARA::PlugIn::AudioSource* audioSource) noexcept
{
    delete audioSource;
}

void MambaDocumentController::doUpdateAudioSourceContent(ARA::PlugIn::AudioSource*, const ARA::ARAContentTimeRange*, ARA::ContentUpdateScopes) noexcept
{
}

void MambaDocumentController::willEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource*, bool) noexcept
{
    audioSourceAccessEnabled_ = true;
}

void MambaDocumentController::didEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource*, bool) noexcept
{
}

ARA::PlugIn::AudioModification* MambaDocumentController::doCreateAudioModification(ARA::PlugIn::AudioSource* audioSource, ARA::ARAAudioModificationHostRef hostRef, const ARA::PlugIn::AudioModification* optionalModificationToClone) noexcept
{
    return new ARA::PlugIn::AudioModification(audioSource, hostRef, optionalModificationToClone);
}

void MambaDocumentController::doDestroyAudioModification(ARA::PlugIn::AudioModification* audioModification) noexcept
{
    delete audioModification;
}

ARA::PlugIn::PlaybackRegion* MambaDocumentController::doCreatePlaybackRegion(ARA::PlugIn::AudioModification* modification, ARA::ARAPlaybackRegionHostRef hostRef) noexcept
{
    return new ARA::PlugIn::PlaybackRegion(modification, hostRef);
}

void MambaDocumentController::doDestroyPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept
{
    delete playbackRegion;
}

ARA::PlugIn::PlaybackRenderer* MambaDocumentController::doCreatePlaybackRenderer() noexcept
{
    return new MambaPlaybackRenderer(this);
}

void MambaDocumentController::doDestroyPlaybackRenderer(ARA::PlugIn::PlaybackRenderer* playbackRenderer) noexcept
{
    delete playbackRenderer;
}

ARA::PlugIn::EditorRenderer* MambaDocumentController::doCreateEditorRenderer() noexcept
{
    return nullptr;
}

void MambaDocumentController::doDestroyEditorRenderer(ARA::PlugIn::EditorRenderer*) noexcept
{
}

ARA::PlugIn::EditorView* MambaDocumentController::doCreateEditorView() noexcept
{
    return nullptr;
}

void MambaDocumentController::doDestroyEditorView(ARA::PlugIn::EditorView*) noexcept
{
}

bool MambaDocumentController::doRestoreObjectsFromArchive(ARA::PlugIn::HostArchiveReader* archiveReader, const ARA::PlugIn::RestoreObjectsFilter*) noexcept
{
    if (!archiveReader) return false;

    struct ArchivedParams {
        float bypass;
        float wetDry;
        float intensity;
    };

    ARA::ARASize archiveSize = archiveReader->getArchiveSize();
    if (archiveSize < sizeof(ArchivedParams)) return false;

    ArchivedParams params{};
    if (!archiveReader->readBytesFromArchive(0, sizeof(params), reinterpret_cast<ARA::ARAByte*>(&params))) {
        return false;
    }

    bypass.store(params.bypass, std::memory_order_release);
    wetDry.store(params.wetDry, std::memory_order_release);
    intensity.store(params.intensity, std::memory_order_release);

    return true;
}

bool MambaDocumentController::doStoreObjectsToArchive(ARA::PlugIn::HostArchiveWriter* archiveWriter, const ARA::PlugIn::StoreObjectsFilter*) noexcept
{
    if (!archiveWriter) return false;

    struct ArchivedParams {
        float bypass;
        float wetDry;
        float intensity;
    };

    ArchivedParams params;
    params.bypass = bypass.load(std::memory_order_acquire);
    params.wetDry = wetDry.load(std::memory_order_acquire);
    params.intensity = intensity.load(std::memory_order_acquire);

    return archiveWriter->writeBytesToArchive(0, sizeof(params), reinterpret_cast<const ARA::ARAByte*>(&params));
}

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

const char* MambaFactoryConfig::getFactoryID() const noexcept
{
    return "com.mamba.audio.suite.ara";
}

const char* MambaFactoryConfig::getPlugInName() const noexcept
{
    return "Mamba-SSM Processor (ARA)";
}

const char* MambaFactoryConfig::getManufacturerName() const noexcept
{
    return "Elite DSP";
}

const char* MambaFactoryConfig::getInformationURL() const noexcept
{
    return "https://example.com/mamba_audio_suite";
}

const char* MambaFactoryConfig::getVersion() const noexcept
{
    return "1.0.0";
}

const char* MambaFactoryConfig::getDocumentArchiveID() const noexcept
{
    return "com.mamba.audio.suite.archive";
}

ARA::PlugIn::DocumentController* MambaFactoryConfig::createDocumentController(const ARA::PlugIn::PlugInEntry* entry, const ARA::ARADocumentControllerHostInstance* instance) const noexcept
{
    return new MambaDocumentController(entry, instance);
}

const ARA::ARAFactory* getMambaARAFactory() noexcept
{
    return ARA::PlugIn::PlugInEntry::getPlugInEntry<MambaFactoryConfig, MambaDocumentController>()->getFactory();
}

extern "C" const ARA::ARAFactory* ARAGetFactory()
{
    return getMambaARAFactory();
}
