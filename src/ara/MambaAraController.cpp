#include "MambaAraController.hpp"
#include "core/Platform.hpp"
#include "core/ConditioningGenerator.hpp"

// -----------------------------------------------------------------------------
// PlaybackRenderer
// -----------------------------------------------------------------------------

MambaPlaybackRenderer::MambaPlaybackRenderer(ARA::PlugIn::DocumentController* documentController)
: ARA::PlugIn::PlaybackRenderer(documentController)
{
    MambaInferenceEngine::Config config;
    config.modelPath = mambaResolveModulePath((void*)&ARAGetFactory, "mamba_rt.onnx");
    config.maxBlockSize = 1024;
    // Zero projection weights — no conditioning effect until model training.
    // CLARA-09 will replace these with trained weights.
    config.deltaWeights.assign(ConditioningFrame::kCondDim, 0.0f);
    config.bWeights.assign(ConditioningFrame::kCondDim, 0.0f);
    rtEngine_ = std::make_unique<MambaInferenceEngine>(config);
    if (!rtEngine_->init()) {
        // Engine init failed — keep the controller usable but log the error.
        // The engine pointer will remain nullptr in the document controller
        // so doUpdateMusicalContextContent() will be a no-op.
        std::fprintf(stderr, "[CLARA] ARA: PlaybackRenderer engine init failed — conditioning disabled\n");
        rtEngine_.reset();
        return;
    }

    // Give the document controller access to the engine so
    // doUpdateMusicalContextContent() can push conditioning updates.
    auto* docCtrl = static_cast<MambaDocumentController*>(documentController);
    docCtrl->setAraEngine(rtEngine_.get());
}

MambaPlaybackRenderer::~MambaPlaybackRenderer() noexcept = default;

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

void MambaDocumentController::doUpdateMusicalContextContent(ARA::PlugIn::MusicalContext* context, const ARA::ARAContentTimeRange* range, ARA::ContentUpdateScopes scopeFlags) noexcept
{
    // ARA musical context (tempo, key, time signature) changed.
    // The engine pointer is wired and setConditioning() is ready.
    //
    // TODO(CLARA-09): Extract actual context via ARA host content reader:
    //   1. Check content availability via host content access controller
    //   2. Create content reader for ARAContentType::kMusicalContext
    //   3. Read tempo, key pitch class, key mode from content events
    //   4. Encode to cond vector via ConditioningGenerator::encode()
    //   5. Call engine->setConditioning()
    //
    // For now, push a neutral (all-zeros) conditioning to prove the path.
    // This effectively disables conditioning until CLARA-09 delivers
    // the host bridge with proper ARA content reading.
    auto* engine = araEngine_;
    if (engine) {
        float neutral[ConditioningFrame::kCondDim];
        ConditioningGenerator::neutral(neutral, ConditioningFrame::kCondDim);
        engine->setConditioning(neutral, ConditioningFrame::kCondDim);
    }

    contentDirty_.store(true, std::memory_order_release);
    (void)context; (void)range; (void)scopeFlags;
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

void MambaDocumentController::doUpdateAudioSourceContent(ARA::PlugIn::AudioSource* source, const ARA::ARAContentTimeRange* range, ARA::ContentUpdateScopes scopeFlags) noexcept
{
    contentDirty_.store(true, std::memory_order_release);
    (void)source; (void)range; (void)scopeFlags;
}

void MambaDocumentController::doAnalyzeRegion(ARA::PlugIn::AudioSource*, ARA::ARAAudioModificationHostRef,
    const ARA::ARAContentTimeRange*, ARA::ContentUpdateScopes) noexcept
{
    // Stub — analysis pipeline not yet implemented.
    // Future: when ARA host bridge is built, feed audio samples
    // through the inference engine and write content (tempo, key)
    // back to the host via ContentUpdateScopes.
}

void MambaDocumentController::willEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource* source, bool enable) noexcept
{
    // When the host is about to grant sample access, prepare internal buffers.
    // Future: pre-allocate analysis buffers or increment access ref count.
    (void)source;
    (void)enable;
}

void MambaDocumentController::didEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource* source, bool enable) noexcept
{
    // The host has granted (or revoked) sample access.
    // When access is enabled, mark content dirty so it can be re-analyzed.
    // When disabled, clean up analysis state.
    if (enable) {
        contentDirty_.store(true, std::memory_order_release);
    }
    (void)source;
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
    return new MambaEditorRenderer(this);
}

void MambaDocumentController::doDestroyEditorRenderer(ARA::PlugIn::EditorRenderer* editorRenderer) noexcept
{
    if (editorRenderer) {
        delete static_cast<MambaEditorRenderer*>(editorRenderer);
    }
}

ARA::PlugIn::EditorView* MambaDocumentController::doCreateEditorView() noexcept
{
    return new MambaEditorView(this);
}

void MambaDocumentController::doDestroyEditorView(ARA::PlugIn::EditorView* editorView) noexcept
{
    if (editorView) {
        delete static_cast<MambaEditorView*>(editorView);
    }
}

// Archive versioning: magic number + version byte permit forward-compatible
// format evolution without corrupting old projects.
static constexpr uint32_t kArchiveMagic = 0x4D424152;  // "MBAR"
static constexpr uint32_t kArchiveVersion = 1;

#pragma pack(push, 1)
struct ArchiveHeader {
    uint32_t magic;
    uint32_t version;
};
struct ArchivedParams {
    float bypass;
    float wetDry;
    float intensity;
};
#pragma pack(pop)

bool MambaDocumentController::doRestoreObjectsFromArchive(ARA::PlugIn::HostArchiveReader* archiveReader, const ARA::PlugIn::RestoreObjectsFilter*) noexcept
{
    if (!archiveReader) return false;

    ARA::ARASize archiveSize = archiveReader->getArchiveSize();
    if (archiveSize < sizeof(ArchiveHeader) + sizeof(ArchivedParams)) return false;

    ArchiveHeader header{};
    if (!archiveReader->readBytesFromArchive(0, sizeof(header), reinterpret_cast<ARA::ARAByte*>(&header))) {
        return false;
    }

    if (header.magic != kArchiveMagic) {
        std::fprintf(stderr, "[CLARA] ARA: archive magic mismatch — expected 0x%08X, got 0x%08X\n",
                     kArchiveMagic, header.magic);
        return false;
    }

    if (header.version > kArchiveVersion) {
        std::fprintf(stderr, "[CLARA] ARA: archive version %u > supported %u — rejecting\n",
                     header.version, kArchiveVersion);
        return false;
    }

    ArchivedParams params{};
    if (!archiveReader->readBytesFromArchive(sizeof(header), sizeof(params),
                                              reinterpret_cast<ARA::ARAByte*>(&params))) {
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

    ArchiveHeader header{kArchiveMagic, kArchiveVersion};
    if (!archiveWriter->writeBytesToArchive(0, sizeof(header),
                                             reinterpret_cast<const ARA::ARAByte*>(&header))) {
        return false;
    }

    ArchivedParams params;
    params.bypass = bypass.load(std::memory_order_acquire);
    params.wetDry = wetDry.load(std::memory_order_acquire);
    params.intensity = intensity.load(std::memory_order_acquire);

    return archiveWriter->writeBytesToArchive(sizeof(header), sizeof(params),
                                               reinterpret_cast<const ARA::ARAByte*>(&params));
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
    return MAMBA_PROJECT_URL;
}

const char* MambaFactoryConfig::getVersion() const noexcept
{
    return "1.1.0";
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
